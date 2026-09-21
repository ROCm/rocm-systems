// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cdna4_to_cdna3_lds_hip_test.cpp
/// @brief Small HIP kernels that force rocjitsu's CDNA4->CDNA3 virtual-LDS path.

#include <cstdint>
#include <hip/hip_runtime.h>
#include <vector>

#include <gtest/gtest.h>

namespace {

constexpr int kBlockSize = 256;
constexpr int kTrB16BlockSize = 128;
constexpr int kBlocks = 64;
constexpr int kMioShapeBlockSize = 512;
constexpr int kMioShapeBlocks = 256;
constexpr int kSharedWords = 17024;
constexpr int kExactStaticWords = 16 * 1024;
constexpr int kExactStaticBytes = kExactStaticWords * static_cast<int>(sizeof(uint32_t));
constexpr int kHighBase = kSharedWords - kBlockSize;
constexpr int kHighVectorBase = kSharedWords - kBlockSize * 4;
constexpr int kLargeDsByteOffset = 0x1400;
static_assert(kSharedWords * static_cast<int>(sizeof(uint32_t)) > 64 * 1024);
static_assert(kExactStaticBytes == 64 * 1024);

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    ASSERT_EQ(err, hipSuccess) << "HIP error: " << hipGetErrorString(err);                         \
  } while (0)

using U32x4 = uint32_t __attribute__((ext_vector_type(4)));
using U32x2 = uint32_t __attribute__((ext_vector_type(2)));

constexpr uint32_t pack_u16_pair(uint32_t lo, uint32_t hi) {
  return (lo & 0xffffu) | ((hi & 0xffffu) << 16);
}

constexpr uint32_t tr_b16_halfword_value(int thread, int halfword) {
  return (0x1200u + static_cast<uint32_t>(thread) * 0x11u + static_cast<uint32_t>(halfword)) &
         0xffffu;
}

__global__ void lds_copy_low_kernel(const uint32_t *__restrict__ in, uint32_t *__restrict__ out) {
  __shared__ volatile uint32_t lds[kSharedWords];
  const int tid = threadIdx.x;
  const uint32_t addr =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds)) + tid * sizeof(uint32_t);
  uint32_t value = in[tid];

  // HIP C++ shared-memory accesses may compile to flat LDS instructions through
  // src_shared_base. The current virtual-LDS implementation rewrites DS
  // opcodes, so these tiny tests use inline DS instructions deliberately.
  asm volatile("ds_write_b32 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               :
               : "v"(addr), "v"(value)
               : "memory");
  __syncthreads();
  asm volatile("ds_read_b32 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=v"(value)
               : "v"(addr)
               : "memory");
  out[tid] = value;
}

__global__ void lds_reverse_high_kernel(const uint32_t *__restrict__ in,
                                        uint32_t *__restrict__ out) {
  __shared__ volatile uint32_t lds[kSharedWords];
  const int tid = threadIdx.x;
  const uint32_t base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds));
  const uint32_t store_addr = base + (kHighBase + tid) * sizeof(uint32_t);
  const uint32_t load_addr = base + (kHighBase + (kBlockSize - 1 - tid)) * sizeof(uint32_t);
  uint32_t value = in[tid];

  // Touch the end of a >64 KiB static LDS allocation. This is the smallest
  // check that the translated DS offset is based on the virtual backing buffer,
  // not the host hardware LDS aperture.
  asm volatile("ds_write_b32 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               :
               : "v"(store_addr), "v"(value)
               : "memory");
  __syncthreads();
  asm volatile("ds_read_b32 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=v"(value)
               : "v"(load_addr)
               : "memory");
  out[tid] = value;
}

__global__ void lds_copy_high_kernel(const uint32_t *__restrict__ in, uint32_t *__restrict__ out) {
  __shared__ volatile uint32_t lds[kSharedWords];
  const int tid = threadIdx.x;
  const uint32_t addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds)) +
                        (kHighBase + tid) * sizeof(uint32_t);
  uint32_t value = in[tid];

  // Keep the first high-offset virtual-LDS check intentionally boring: each
  // lane reads back the word it wrote. Reverse and cross-wave tests below then
  // isolate producer/consumer ordering once this baseline is known-good.
  asm volatile("ds_write_b32 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               :
               : "v"(addr), "v"(value)
               : "memory");
  __syncthreads();
  asm volatile("ds_read_b32 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=v"(value)
               : "v"(addr)
               : "memory");
  out[tid] = value;
}

__global__ void lds_copy_high_touch_s100_kernel(const uint32_t *__restrict__ in,
                                                uint32_t *__restrict__ out) {
  __shared__ volatile uint32_t lds[kSharedWords];
  const int tid = threadIdx.x;
  const uint32_t addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds)) +
                        (kHighBase + tid) * sizeof(uint32_t);
  uint32_t value = in[tid];

  // Force the translated source body to name a high ordinary SGPR. Virtual-LDS
  // lowering must then use its spill-per-use path and issue VMEM with a high
  // borrowed SADDR pair, matching the shape of the reduced MIOpen fault.
  asm volatile("s_mov_b32 s100, 0\n" ::: "memory");
  asm volatile("ds_write_b32 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               :
               : "v"(addr), "v"(value)
               : "memory");
  __syncthreads();
  asm volatile("ds_read_b32 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=v"(value)
               : "v"(addr)
               : "memory");
  out[tid] = value;
}

__global__ void lds_multi_block_high_kernel(const uint32_t *__restrict__ in,
                                            uint32_t *__restrict__ out) {
  __shared__ volatile uint32_t lds[kSharedWords];
  const int tid = threadIdx.x;
  const int index = blockIdx.x * blockDim.x + tid;
  const uint32_t addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds)) +
                        (kHighBase + tid) * sizeof(uint32_t);
  uint32_t value = in[index];

  // Every workgroup writes the same high LDS offsets. Virtual LDS must offset
  // the backing pointer by workgroup id so concurrent workgroups do not alias.
  asm volatile("ds_write_b32 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               :
               : "v"(addr), "v"(value)
               : "memory");
  __syncthreads();
  asm volatile("ds_read_b32 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=v"(value)
               : "v"(addr)
               : "memory");
  out[index] = value;
}

__global__ void lds_cross_wave_exchange_high_kernel(const uint32_t *__restrict__ in,
                                                    uint32_t *__restrict__ out) {
  __shared__ volatile uint32_t lds[kSharedWords];
  const int tid = threadIdx.x;
  const uint32_t base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds));
  const uint32_t store_addr = base + (kHighBase + tid) * sizeof(uint32_t);
  const uint32_t load_addr = base + (kHighBase + (tid ^ 64)) * sizeof(uint32_t);
  uint32_t value = in[tid];

  // This is the smallest LDS producer/consumer check that forces traffic across
  // wavefronts. Matmul-style tiles rely on one wave observing data another wave
  // wrote before the block barrier, so a virtual-LDS global-memory replacement
  // must preserve both the data and the synchronization semantics here.
  asm volatile("ds_write_b32 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               :
               : "v"(store_addr), "v"(value)
               : "memory");
  __syncthreads();
  asm volatile("ds_read_b32 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=v"(value)
               : "v"(load_addr)
               : "memory");
  out[tid] = value;
}

__global__ void lds_b128_reverse_high_kernel(const uint32_t *__restrict__ in,
                                             uint32_t *__restrict__ out) {
  __shared__ volatile uint32_t lds[kSharedWords];
  const int tid = threadIdx.x;
  const int source = tid * 4;
  const int target = (kBlockSize - 1 - tid) * 4;
  const uint32_t base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds));
  const uint32_t store_addr = base + (kHighVectorBase + source) * sizeof(uint32_t);
  const uint32_t load_addr = base + (kHighVectorBase + target) * sizeof(uint32_t);
  U32x4 value = {in[source + 0], in[source + 1], in[source + 2], in[source + 3]};

  // The fp16 Tensile repro uses b128 LDS traffic. Keep this fixture as small as
  // possible while covering the same vector-width lowering path.
  asm volatile("ds_write_b128 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               :
               : "v"(store_addr), "v"(value)
               : "memory");
  __syncthreads();
  asm volatile("ds_read_b128 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=v"(value)
               : "v"(load_addr)
               : "memory");

  out[source + 0] = value[0];
  out[source + 1] = value[1];
  out[source + 2] = value[2];
  out[source + 3] = value[3];
}

__global__ void lds_b128_reverse_immediate_offset_kernel(const uint32_t *__restrict__ in,
                                                         uint32_t *__restrict__ out) {
  __shared__ volatile uint32_t lds[kSharedWords];
  const int tid = threadIdx.x;
  const int source = tid * 4;
  const int target = (kBlockSize - 1 - tid) * 4;
  const uint32_t base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds));
  const uint32_t store_addr = base + source * sizeof(uint32_t);
  const uint32_t load_addr = base + target * sizeof(uint32_t);
  U32x4 value = {in[source + 0], in[source + 1], in[source + 2], in[source + 3]};

  static_assert(kLargeDsByteOffset + kBlockSize * 4 * static_cast<int>(sizeof(uint32_t)) <=
                kSharedWords * static_cast<int>(sizeof(uint32_t)));

  // Tensile kernels commonly keep a small vector LDS address in a VGPR and put
  // the tile displacement in the DS instruction's immediate offset. When that
  // offset is larger than CDNA3 global-memory's immediate range, virtual-LDS
  // lowering must materialize an address in a scratch VGPR and restore every
  // clobbered register afterwards.
  asm volatile("ds_write_b128 %0, %1 offset:5120\n"
               "s_waitcnt lgkmcnt(0)\n"
               :
               : "v"(store_addr), "v"(value)
               : "memory");
  __syncthreads();
  asm volatile("ds_read_b128 %0, %1 offset:5120\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=v"(value)
               : "v"(load_addr)
               : "memory");

  out[source + 0] = value[0];
  out[source + 1] = value[1];
  out[source + 2] = value[2];
  out[source + 3] = value[3];
}

__global__ void lds_read_b64_tr_b16_kernel(uint32_t *__restrict__ out) {
  __shared__ volatile uint32_t lds[kSharedWords];
  const int tid = threadIdx.x;
  const uint32_t base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds));
  const uint32_t addr = base + tid * sizeof(U32x2);
  U32x2 raw = {pack_u16_pair(tr_b16_halfword_value(tid, 0), tr_b16_halfword_value(tid, 1)),
               pack_u16_pair(tr_b16_halfword_value(tid, 2), tr_b16_halfword_value(tid, 3))};
  U32x2 value = {};

  // `ds_read_b64_tr_b16` is a distinct Tensile-shaped LDS read: it reads a
  // per-lane b64 footprint and returns a 4x16-lane halfword transpose through
  // the DS crossbar. Keep this fixture all-lane and full-wave, matching the ISA
  // contract assumed by the lowering.
  asm volatile("ds_write_b64 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               :
               : "v"(addr), "v"(raw)
               : "memory");
  __syncthreads();
  asm volatile("ds_read_b64_tr_b16 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=v"(value)
               : "v"(addr)
               : "memory");

  out[tid * 2 + 0] = value[0];
  out[tid * 2 + 1] = value[1];
}

__global__ void lds_zero_kernarg_dispatch_ptr_kernel() {
  __shared__ volatile uint32_t lds[kSharedWords];
  const int tid = threadIdx.x;
  const uint32_t addr =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds)) + tid * sizeof(uint32_t);
  const uintptr_t dispatch_ptr = reinterpret_cast<uintptr_t>(__builtin_amdgcn_dispatch_ptr());
  const uint32_t dispatch_ptr_lo = static_cast<uint32_t>(dispatch_ptr);
  uint32_t value = 0x7a5a0000u ^ static_cast<uint32_t>(tid * 0x101u);

  // This kernel intentionally has no formal kernargs but consumes the
  // dispatch-packet pointer. Virtual LDS must still pass its runtime state
  // through a DBT-owned wrapper kernarg and leave the packet fields guest-owned.
  asm volatile("; keep dispatch ptr live for descriptor selection: %0"
               :
               : "s"(dispatch_ptr_lo)
               : "memory");
  if (dispatch_ptr == 0)
    value ^= 0xdead0000u;
  asm volatile("ds_write_b32 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               :
               : "v"(addr), "v"(value)
               : "memory");
  __syncthreads();
  asm volatile("ds_read_b32 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=v"(value)
               : "v"(addr)
               : "memory");
  asm volatile("; keep virtual LDS read live: %0" : : "v"(value) : "memory");
}

template <bool TouchHighSgpr> __global__ void lds_static64_dynamic64_zero_kernarg_kernel() {
  __shared__ volatile uint32_t static_lds[kExactStaticWords];
  extern __shared__ uint32_t dynamic_lds[];
  const int tid = threadIdx.x;
  const uintptr_t dispatch_ptr = reinterpret_cast<uintptr_t>(__builtin_amdgcn_dispatch_ptr());
  const uint32_t dispatch_ptr_lo = static_cast<uint32_t>(dispatch_ptr);
  const uint32_t static_addr =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(static_lds)) + tid * sizeof(uint32_t);
  const uint32_t dynamic_addr =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dynamic_lds)) + tid * sizeof(uint32_t);
  const uint32_t static_imm_addr = static_addr + 0x4000u;
  const uint32_t dynamic_imm_addr = dynamic_addr + 0x4000u;
  uint32_t cold_static_read = 0;
  uint32_t cold_dynamic_read = 0;
  uint32_t static_imm_read = 0;
  uint32_t dynamic_imm_read = 0;

  // MIOpen's reduced stride2 convolution uses a zero-formal-kernarg descriptor
  // with the dispatch-packet pointer enabled, 64 KiB fixed LDS, and packet LDS.
  // Keep this fixture output-free so HIP emits the same zero-formal-kernarg
  // descriptor shape while still executing the virtualized cold-read +
  // immediate-offset DS accesses that fault in the real kernel.
  asm volatile("; keep dispatch ptr live for descriptor selection: %0"
               :
               : "s"(dispatch_ptr_lo)
               : "memory");
  if constexpr (TouchHighSgpr) {
    asm volatile("s_mov_b32 s100, 0\n" ::: "memory");
  }
  asm volatile("ds_read_b32 %0, %2\n"
               "ds_read_b32 %1, %3\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=v"(cold_static_read), "=v"(cold_dynamic_read)
               : "v"(static_addr), "v"(dynamic_addr)
               : "memory");
  asm volatile("ds_write_b32 %0, %1 offset:256\n"
               "ds_write_b32 %2, %3 offset:256\n"
               "s_waitcnt lgkmcnt(0)\n"
               :
               : "v"(static_imm_addr), "v"(cold_static_read), "v"(dynamic_imm_addr),
                 "v"(cold_dynamic_read)
               : "memory");
  __syncthreads();
  asm volatile("ds_read_b32 %0, %2 offset:256\n"
               "ds_read_b32 %1, %3 offset:256\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=v"(static_imm_read), "=v"(dynamic_imm_read)
               : "v"(static_imm_addr), "v"(dynamic_imm_addr)
               : "memory");
  asm volatile("; keep zero-kernarg static+dynamic virtual LDS reads live: %0 %1"
               :
               : "v"(static_imm_read), "v"(dynamic_imm_read)
               : "memory");
}

template <bool TouchHighSgpr> __global__ void lds_miopen_c220_zero_kernarg_kernel() {
  __shared__ volatile uint32_t static_lds[kExactStaticWords];
  extern __shared__ uint32_t dynamic_lds[];
  const uintptr_t dispatch_ptr = reinterpret_cast<uintptr_t>(__builtin_amdgcn_dispatch_ptr());
  const uint32_t dispatch_ptr_lo = static_cast<uint32_t>(dispatch_ptr);
  const uint32_t static_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(static_lds));
  const uint32_t dynamic_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dynamic_lds));
  const uint32_t lane_addr = 0xc220u + (static_cast<uint32_t>(threadIdx.x) & 3u) * 4u;
  const uint32_t value = static_cast<uint32_t>(blockIdx.x) ^ lane_addr;
  uint32_t read = 0;

  // This mirrors the reduced MIOpen faulting virtual-LDS access as closely as a
  // small HIP fixture can: the real kernel reaches
  //   ds_read_b32 v116, v62 offset:256
  // with v62 cycling through 0xc220, 0xc224, 0xc228, and 0xc22c under the
  // zero-formal-kernarg dispatch-packet-pointer ABI.
  asm volatile("; keep dispatch ptr live for descriptor selection: %0"
               :
               : "s"(dispatch_ptr_lo)
               : "memory");
  asm volatile("; keep static LDS allocation live for descriptor sizing: %0"
               :
               : "v"(static_addr)
               : "memory");
  asm volatile("; keep dynamic LDS live for packet LDS sizing: %0"
               :
               : "v"(dynamic_addr)
               : "memory");
  if constexpr (TouchHighSgpr) {
    asm volatile("s_mov_b32 s100, 0\n" ::: "memory");
  }
  asm volatile("ds_write_b32 %0, %1 offset:256\n"
               "s_waitcnt lgkmcnt(0)\n"
               :
               : "v"(lane_addr), "v"(value)
               : "memory");
  __syncthreads();
  asm volatile("ds_read_b32 %0, %1 offset:256\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=v"(read)
               : "v"(lane_addr)
               : "memory");
  asm volatile("; keep MIOpen-shaped c220 LDS read live: %0" : : "v"(read) : "memory");
}

template <bool TouchHighSgpr> __global__ void lds_miopen_c220_v116_zero_kernarg_kernel() {
  __shared__ volatile uint32_t static_lds[kExactStaticWords];
  extern __shared__ uint32_t dynamic_lds[];
  const uintptr_t dispatch_ptr = reinterpret_cast<uintptr_t>(__builtin_amdgcn_dispatch_ptr());
  const uint32_t dispatch_ptr_lo = static_cast<uint32_t>(dispatch_ptr);
  const uint32_t static_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(static_lds));
  const uint32_t dynamic_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dynamic_lds));
  const uint32_t lane_addr = 0xc220u + (static_cast<uint32_t>(threadIdx.x) & 3u) * 4u;
  const uint32_t value = static_cast<uint32_t>(blockIdx.x) ^ lane_addr;

  // Match the exact fixed destination register from the reduced MIOpen fault:
  //   ds_read_b32 v116, v62 offset:256
  // The fixed-register operands keep the source descriptor and the virtual-LDS
  // lowering honest about high VGPR use without depending on compiler register
  // allocation choices in this small fixture.
  asm volatile("; keep dispatch ptr live for descriptor selection: %0"
               :
               : "s"(dispatch_ptr_lo)
               : "memory");
  asm volatile("; keep static LDS allocation live for descriptor sizing: %0"
               :
               : "v"(static_addr)
               : "memory");
  asm volatile("; keep dynamic LDS live for packet LDS sizing: %0"
               :
               : "v"(dynamic_addr)
               : "memory");
  if constexpr (TouchHighSgpr) {
    asm volatile("s_mov_b32 s100, 0\n" ::: "memory");
  }
  asm volatile("v_mov_b32 v62, %0\n"
               "ds_write_b32 v62, %1 offset:256\n"
               "s_waitcnt lgkmcnt(0)\n"
               "s_barrier\n"
               "ds_read_b32 v116, v62 offset:256\n"
               "s_waitcnt lgkmcnt(0)\n"
               :
               : "v"(lane_addr), "v"(value)
               : "memory");
}

template <bool TouchHighSgpr>
__global__ void lds_static64_dynamic64_dispatch_ptr_kernel(uint32_t *__restrict__ status_out) {
  __shared__ volatile uint32_t static_lds[kExactStaticWords];
  extern __shared__ uint32_t dynamic_lds[];
  const int tid = threadIdx.x;
  const int index = blockIdx.x * blockDim.x + tid;
  const uintptr_t dispatch_ptr = reinterpret_cast<uintptr_t>(__builtin_amdgcn_dispatch_ptr());
  const uint32_t dispatch_ptr_lo = static_cast<uint32_t>(dispatch_ptr);
  const uint32_t static_addr =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(static_lds)) + tid * sizeof(uint32_t);
  const uint32_t dynamic_addr =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dynamic_lds)) + tid * sizeof(uint32_t);
  const uint32_t static_imm_addr = static_addr + 0x4000u;
  const uint32_t dynamic_imm_addr = dynamic_addr + 0x4000u;
  const uint32_t static_value = 0x51000000u ^ static_cast<uint32_t>(index * 0x101u);
  const uint32_t dynamic_value = 0xa6000000u ^ static_cast<uint32_t>(index * 0x181u);
  const uint32_t static_imm_value = 0x72000000u ^ static_cast<uint32_t>(index * 0x105u);
  const uint32_t dynamic_imm_value = 0xc8000000u ^ static_cast<uint32_t>(index * 0x185u);
  uint32_t cold_static_read = 0;
  uint32_t cold_dynamic_read = 0;
  uint32_t static_read = 0;
  uint32_t dynamic_read = 0;
  uint32_t static_imm_read = 0;
  uint32_t dynamic_imm_read = 0;

  // Match the reduced MIOpen LDS resource shape: the descriptor contributes
  // exactly 64 KiB of fixed LDS and the AQL packet contributes additional LDS
  // at dispatch time.
  asm volatile("; keep dispatch ptr live for descriptor selection: %0"
               :
               : "s"(dispatch_ptr_lo)
               : "memory");
  if constexpr (TouchHighSgpr) {
    // Keep this variant close to the reduced MIOpen fault. Naming s100 forces
    // the descriptor near the ordinary SGPR limit, so virtual-LDS lowering must
    // borrow a high SADDR pair and spill/reload the virtual base around each
    // rewritten DS operation.
    asm volatile("s_mov_b32 s100, 0\n" ::: "memory");
  }
  // LDS reads before a same-address write are undefined but legal. MIOpen uses
  // this shape on the reduced fp16 conv path, so virtual LDS must not rely on a
  // backing location having been written before the first lowered global load.
  asm volatile("ds_read_b32 %0, %2\n"
               "ds_read_b32 %1, %3\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=v"(cold_static_read), "=v"(cold_dynamic_read)
               : "v"(static_addr), "v"(dynamic_addr)
               : "memory");
  asm volatile("; keep cold virtual-LDS reads live: %0 %1"
               :
               : "v"(cold_static_read), "v"(cold_dynamic_read)
               : "memory");
  asm volatile("ds_write_b32 %0, %1\n"
               "ds_write_b32 %2, %3\n"
               "ds_write_b32 %4, %5 offset:256\n"
               "ds_write_b32 %6, %7 offset:256\n"
               "s_waitcnt lgkmcnt(0)\n"
               :
               : "v"(static_addr), "v"(static_value), "v"(dynamic_addr), "v"(dynamic_value),
                 "v"(static_imm_addr), "v"(static_imm_value), "v"(dynamic_imm_addr),
                 "v"(dynamic_imm_value)
               : "memory");
  __syncthreads();
  asm volatile("ds_read_b32 %0, %4\n"
               "ds_read_b32 %1, %5\n"
               "ds_read_b32 %2, %6 offset:256\n"
               "ds_read_b32 %3, %7 offset:256\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=v"(static_read), "=v"(dynamic_read), "=v"(static_imm_read),
                 "=v"(dynamic_imm_read)
               : "v"(static_addr), "v"(dynamic_addr), "v"(static_imm_addr), "v"(dynamic_imm_addr)
               : "memory");

  uint32_t status = 0;
  if (dispatch_ptr == 0)
    status |= 1u;
  if (static_read != static_value)
    status |= 2u;
  if (dynamic_read != dynamic_value)
    status |= 4u;
  if (static_imm_read != static_imm_value)
    status |= 8u;
  if (dynamic_imm_read != dynamic_imm_value)
    status |= 16u;
  status_out[index] = status;
}

std::vector<uint32_t> make_input(size_t count, uint32_t salt) {
  std::vector<uint32_t> values(count);
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = 0x9e370000u ^ salt ^ static_cast<uint32_t>(i * 2654435761u);
  return values;
}

void run_unary_kernel(void (*kernel)(const uint32_t *, uint32_t *),
                      const std::vector<uint32_t> &expected_input,
                      const std::vector<uint32_t> &expected_output) {
  ASSERT_EQ(expected_input.size(), expected_output.size());
  const size_t bytes = expected_input.size() * sizeof(uint32_t);

  uint32_t *in = nullptr;
  uint32_t *out = nullptr;
  HIP_ASSERT(hipMalloc(&in, bytes));
  HIP_ASSERT(hipMalloc(&out, bytes));
  HIP_ASSERT(hipMemcpy(in, expected_input.data(), bytes, hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemset(out, 0x5a, bytes));

  kernel<<<expected_input.size() / kBlockSize, kBlockSize>>>(in, out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint32_t> actual(expected_output.size());
  HIP_ASSERT(hipMemcpy(actual.data(), out, bytes, hipMemcpyDeviceToHost));
  (void)hipFree(in);
  (void)hipFree(out);

  uint32_t mismatches = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] == expected_output[i])
      continue;
    if (mismatches < 8) {
      ADD_FAILURE() << "mismatch at i=" << i << ": got=0x" << std::hex << actual[i]
                    << " expected=0x" << expected_output[i] << std::dec;
    }
    ++mismatches;
  }
  EXPECT_EQ(mismatches, 0u) << mismatches << " mismatches";
}

void run_output_kernel(void (*kernel)(uint32_t *), const std::vector<uint32_t> &expected_output,
                       int threads) {
  const size_t bytes = expected_output.size() * sizeof(uint32_t);

  uint32_t *out = nullptr;
  HIP_ASSERT(hipMalloc(&out, bytes));
  HIP_ASSERT(hipMemset(out, 0x5a, bytes));

  kernel<<<1, threads>>>(out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint32_t> actual(expected_output.size());
  HIP_ASSERT(hipMemcpy(actual.data(), out, bytes, hipMemcpyDeviceToHost));
  (void)hipFree(out);

  uint32_t mismatches = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] == expected_output[i])
      continue;
    if (mismatches < 8) {
      ADD_FAILURE() << "mismatch at i=" << i << ": got=0x" << std::hex << actual[i]
                    << " expected=0x" << expected_output[i] << std::dec;
    }
    ++mismatches;
  }
  EXPECT_EQ(mismatches, 0u) << mismatches << " mismatches";
}

void run_zero_kernarg_kernel() {
  lds_zero_kernarg_dispatch_ptr_kernel<<<kBlocks, kBlockSize>>>();
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
}

template <bool TouchHighSgpr> void run_static_dynamic_zero_kernarg_kernel() {
  lds_static64_dynamic64_zero_kernarg_kernel<TouchHighSgpr>
      <<<kMioShapeBlocks, kMioShapeBlockSize, kExactStaticBytes>>>();
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
}

template <bool TouchHighSgpr> void run_miopen_c220_zero_kernarg_kernel() {
  lds_miopen_c220_zero_kernarg_kernel<TouchHighSgpr>
      <<<kMioShapeBlocks, kMioShapeBlockSize, kExactStaticBytes>>>();
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
}

template <bool TouchHighSgpr> void run_miopen_c220_v116_zero_kernarg_kernel() {
  lds_miopen_c220_v116_zero_kernarg_kernel<TouchHighSgpr>
      <<<kMioShapeBlocks, kMioShapeBlockSize, kExactStaticBytes>>>();
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
}

template <bool TouchHighSgpr> void run_static_dynamic_dispatch_ptr_kernel() {
  constexpr size_t kStatusWords = static_cast<size_t>(kMioShapeBlocks) * kMioShapeBlockSize;
  constexpr size_t kStatusBytes = kStatusWords * sizeof(uint32_t);
  uint32_t *status_device = nullptr;
  HIP_ASSERT(hipMalloc(&status_device, kStatusBytes));
  HIP_ASSERT(hipMemset(status_device, 0x5a, kStatusBytes));

  lds_static64_dynamic64_dispatch_ptr_kernel<TouchHighSgpr>
      <<<kMioShapeBlocks, kMioShapeBlockSize, kExactStaticBytes>>>(status_device);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint32_t> status(kStatusWords);
  HIP_ASSERT(hipMemcpy(status.data(), status_device, kStatusBytes, hipMemcpyDeviceToHost));

  uint32_t failures = 0;
  for (size_t i = 0; i < status.size(); ++i) {
    if (status[i] == 0)
      continue;
    if (failures < 8)
      ADD_FAILURE() << "status failure at i=" << i << ": 0x" << std::hex << status[i] << std::dec;
    ++failures;
  }
  EXPECT_EQ(failures, 0u) << failures << " static+dynamic LDS status failures";
  (void)hipFree(status_device);
}

} // namespace

TEST(HipLdsCopyDbtTest, LargeStaticLdsCopyLowOffset) {
  const std::vector<uint32_t> input = make_input(kBlockSize, 0x101u);
  run_unary_kernel(lds_copy_low_kernel, input, input);
}

TEST(HipLdsCopyDbtTest, LargeStaticLdsReverseHighOffset) {
  const std::vector<uint32_t> input = make_input(kBlockSize, 0x202u);
  std::vector<uint32_t> expected(input.rbegin(), input.rend());
  run_unary_kernel(lds_reverse_high_kernel, input, expected);
}

TEST(HipLdsCopyDbtTest, LargeStaticLdsCopyHighOffset) {
  const std::vector<uint32_t> input = make_input(kBlockSize, 0x707u);
  run_unary_kernel(lds_copy_high_kernel, input, input);
}

TEST(HipLdsCopyDbtTest, LargeStaticLdsCopyHighOffsetWithBorrowedHighSgpr) {
  const std::vector<uint32_t> input = make_input(kBlockSize, 0x808u);
  run_unary_kernel(lds_copy_high_touch_s100_kernel, input, input);
}

TEST(HipLdsCopyDbtTest, LargeStaticLdsMultiBlockHighOffset) {
  const std::vector<uint32_t> input = make_input(kBlocks * kBlockSize, 0x303u);
  run_unary_kernel(lds_multi_block_high_kernel, input, input);
}

TEST(HipLdsCopyDbtTest, LargeStaticLdsCrossWaveExchangeHighOffset) {
  const std::vector<uint32_t> input = make_input(kBlockSize, 0x606u);
  std::vector<uint32_t> expected(input.size());
  for (int tid = 0; tid < kBlockSize; ++tid)
    expected[tid] = input[tid ^ 64];
  run_unary_kernel(lds_cross_wave_exchange_high_kernel, input, expected);
}

TEST(HipLdsCopyDbtTest, LargeStaticLdsB128ReverseHighOffset) {
  const std::vector<uint32_t> input = make_input(kBlockSize * 4, 0x404u);
  std::vector<uint32_t> expected(input.size());
  for (int tid = 0; tid < kBlockSize; ++tid) {
    const int source = tid * 4;
    const int target = (kBlockSize - 1 - tid) * 4;
    for (int lane = 0; lane < 4; ++lane)
      expected[source + lane] = input[target + lane];
  }
  run_unary_kernel(lds_b128_reverse_high_kernel, input, expected);
}

TEST(HipLdsCopyDbtTest, LargeStaticLdsB128ReverseImmediateOffset) {
  const std::vector<uint32_t> input = make_input(kBlockSize * 4, 0x505u);
  std::vector<uint32_t> expected(input.size());
  for (int tid = 0; tid < kBlockSize; ++tid) {
    const int source = tid * 4;
    const int target = (kBlockSize - 1 - tid) * 4;
    for (int lane = 0; lane < 4; ++lane)
      expected[source + lane] = input[target + lane];
  }
  run_unary_kernel(lds_b128_reverse_immediate_offset_kernel, input, expected);
}

TEST(HipLdsCopyDbtTest, LargeStaticLdsReadB64TrB16) {
  std::vector<uint32_t> expected(kTrB16BlockSize * 2);
  for (int tid = 0; tid < kTrB16BlockSize; ++tid) {
    const int wave_base = (tid / 64) * 64;
    const int lane = tid & 63;
    const int halfword = lane & 3;
    const int source_base = wave_base + (lane & 0x30) + ((lane & 0x0c) >> 2);
    const uint32_t h0 = tr_b16_halfword_value(source_base + 0, halfword);
    const uint32_t h1 = tr_b16_halfword_value(source_base + 4, halfword);
    const uint32_t h2 = tr_b16_halfword_value(source_base + 8, halfword);
    const uint32_t h3 = tr_b16_halfword_value(source_base + 12, halfword);
    expected[tid * 2 + 0] = pack_u16_pair(h0, h1);
    expected[tid * 2 + 1] = pack_u16_pair(h2, h3);
  }
  run_output_kernel(lds_read_b64_tr_b16_kernel, expected, kTrB16BlockSize);
}

TEST(HipLdsCopyDbtTest, LargeStaticLdsZeroKernargWrapperState) { run_zero_kernarg_kernel(); }

TEST(HipLdsCopyDbtTest, Static64Dynamic64ZeroKernargWrapperState) {
  run_static_dynamic_zero_kernarg_kernel<false>();
}

TEST(HipLdsCopyDbtTest, Static64Dynamic64ZeroKernargWrapperStateWithBorrowedHighSgpr) {
  run_static_dynamic_zero_kernarg_kernel<true>();
}

TEST(HipLdsCopyDbtTest, MiopenC220ZeroKernargWrapperState) {
  run_miopen_c220_zero_kernarg_kernel<false>();
}

TEST(HipLdsCopyDbtTest, MiopenC220ZeroKernargWrapperStateWithBorrowedHighSgpr) {
  run_miopen_c220_zero_kernarg_kernel<true>();
}

TEST(HipLdsCopyDbtTest, MiopenC220V116ZeroKernargWrapperState) {
  run_miopen_c220_v116_zero_kernarg_kernel<false>();
}

TEST(HipLdsCopyDbtTest, MiopenC220V116ZeroKernargWrapperStateWithBorrowedHighSgpr) {
  run_miopen_c220_v116_zero_kernarg_kernel<true>();
}

TEST(HipLdsCopyDbtTest, Static64Dynamic64DispatchPointerWrapperState) {
  run_static_dynamic_dispatch_ptr_kernel<false>();
}

TEST(HipLdsCopyDbtTest, Static64Dynamic64DispatchPointerWrapperStateWithBorrowedHighSgpr) {
  run_static_dynamic_dispatch_ptr_kernel<true>();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const int rc = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return rc;
}
