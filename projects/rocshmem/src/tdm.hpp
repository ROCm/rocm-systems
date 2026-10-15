/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

// Tensor Data Mover (TDM) API for gfx1250 (MI450): wraps the
// __builtin_amdgcn_tensor_load_to_lds/store_from_lds intrinsics (descriptor
// construction, cache-policy encoding, bulk tile-copy helpers) plus the
// per-workgroup LDS registration backing rocshmem_set_tdm_lds().

#ifndef LIBRARY_SRC_TDM_HPP_
#define LIBRARY_SRC_TDM_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

// ==============================================================================
// FEATURE DETECTION
// ==============================================================================

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if defined(__HIP_PLATFORM_AMD__) &&                        \
    (__has_builtin(__builtin_amdgcn_tensor_load_to_lds) &&   \
     __has_builtin(__builtin_amdgcn_tensor_store_from_lds))
#define HIP_HAVE_TDM_INTRINSICS 1
#else
#define HIP_HAVE_TDM_INTRINSICS 0
#endif

namespace rocshmem {
namespace tdm {

// ==============================================================================
// VECTOR TYPES
// ==============================================================================

using u32x4 = uint32_t __attribute__((ext_vector_type(4)));
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
using i32x8 = int32_t __attribute__((ext_vector_type(8)));

// ==============================================================================
// ELEMENT SIZE (GROUP1 data_size field)
// ==============================================================================

enum class DataSize : uint32_t { Bits8 = 0, Bits16 = 1, Bits32 = 2, Bits64 = 3 };

constexpr uint32_t data_size_bytes(DataSize d) {
  return 1u << static_cast<uint32_t>(d);
}

// ==============================================================================
// CACHE POLICY (the `cachepolicy` immediate argument of the TDM intrinsics)
// cachepolicy = th_bits[2:0] | (scope_bits[1:0] << 3);
// ==============================================================================

// Memory scope (cachepolicy bits[4:3]), pre-shifted into place here.
enum class Scope : unsigned {
  Wgp = 0,           // workgroup processor: L1-level coherence (default)
  ShaderEngine = 8,  // GL2/L2-level coherence within one shader engine
  Device = 16,       // all shader engines on this GPU
  System = 24,       // CPU + all GPUs on the node
};

// Temporal hint (cachepolicy bits[2:0]) for tensor_load_to_lds. th=7 is
// undefined for loads, so it's intentionally not a member here.
enum class LoadTemporalHint : unsigned {
  Default = 0,                 // write-allocate, cached in L1 and L2
  NonTemporal = 1,             // TH_LOAD_NT: no L1 allocation
  HwManaged = 2,               // TH_LOAD_HT: hardware-managed temporal hint
  LastUse = 3,                 // TH_LOAD_LU: evict from L1 after this read
  NonTemporalReadThrough = 4,  // TH_LOAD_NT_RT
  ReadThroughNonTemporal = 5,  // TH_LOAD_RT_NT
  NonTemporalHwManaged = 6,    // TH_LOAD_NT_HT
};

// Temporal hint for tensor_store_from_lds. Same bit positions as
// LoadTemporalHint, but th=3/th=7 mean something different for stores.
enum class StoreTemporalHint : unsigned {
  Default = 0,                 // write-back
  NonTemporal = 1,             // TH_STORE_NT: non-temporal write-through
  HwManaged = 2,               // TH_STORE_HT: hardware-managed temporal hint
  WriteBack = 3,               // TH_STORE_WB: explicit write-back
  NonTemporalReadThrough = 4,  // TH_STORE_NT_RT
  ReadThroughNonTemporal = 5,  // TH_STORE_RT_NT
  NonTemporalHwManaged = 6,    // TH_STORE_NT_HT
  NonTemporalWriteBack = 7,    // TH_STORE_NT_WB
};

constexpr int make_cache_policy(LoadTemporalHint th, Scope scope = Scope::Wgp) {
  return static_cast<int>(th) | static_cast<int>(scope);
}
constexpr int make_cache_policy(StoreTemporalHint th, Scope scope = Scope::Wgp) {
  return static_cast<int>(th) | static_cast<int>(scope);
}

// ==============================================================================
// DESCRIPTOR CONSTRUCTION
// ==============================================================================

// GROUP0 (addresses + control) and GROUP1 (tensor layout) for a 2D transfer.
// GROUP2/GROUP3 (3D/gather) aren't modeled; every call site here is 2D.
struct Descriptor2D {
  u32x4 group0{};
  i32x8 group1{};
};

// GROUP0 only (LDS address + 48-bit global VA), split out so callers can
// cheaply rebuild just the addresses of an otherwise-invariant descriptor.
__device__ __forceinline__ u32x4 make_group0(uintptr_t lds_addr, uintptr_t global_addr) {
  return u32x4{
      1u,  // m_count = 1
      static_cast<uint32_t>(lds_addr),
      static_cast<uint32_t>(global_addr & 0xFFFFFFFFu),
      // bits[24:0] = global_addr_hi, bits[31:30] = m_type = 2 (non-gather).
      // m_th is left 0: it's overridden by the intrinsic's cachepolicy
      // argument, and m_is_store is set automatically by the compiler.
      (2u << 30) | static_cast<uint32_t>((global_addr >> 32) & 0x1FFFFFFu),
  };
}

// GROUP1: element size, full-tensor/tile dims, and strides (in elements).
// Strides are genuinely 48-bit here, unlike AMD's own amd_gfx1250_TDM.h
// header, whose tensorDimNStride() setters truncate to 32 bits.
__device__ __forceinline__ i32x8 make_group1(DataSize data_size, uint32_t tensor_dim0,
                                             uint32_t tensor_dim1, uint32_t tile_dim0,
                                             uint32_t tile_dim1, uint64_t tensor_dim0_stride,
                                             uint64_t tensor_dim1_stride,
                                             uint32_t workgroup_mask = 1) {
  uint32_t s[8]{};
  s[0] = (workgroup_mask & 0xFFFFu) | (static_cast<uint32_t>(data_size) << 16);
  s[1] = (tensor_dim0 & 0xFFFFu) << 16;  // atomic_barrier_address left 0
  s[2] = ((tensor_dim0 >> 16) & 0xFFFFu) | ((tensor_dim1 & 0xFFFFu) << 16);
  s[3] = ((tensor_dim1 >> 16) & 0xFFFFu) | ((tile_dim0 & 0xFFFFu) << 16);
  s[4] = (tile_dim1 & 0xFFFFu);  // tile_dim2 left 0 (2D only)
  s[5] = static_cast<uint32_t>(tensor_dim0_stride & 0xFFFFFFFFull);
  s[6] = static_cast<uint32_t>((tensor_dim0_stride >> 32) & 0xFFFFull) |
         (static_cast<uint32_t>(tensor_dim1_stride & 0xFFFFull) << 16);
  s[7] = static_cast<uint32_t>((tensor_dim1_stride >> 16) & 0xFFFFFFFFull);

  i32x8 g1{};
  for (int i = 0; i < 8; ++i) g1[i] = static_cast<int32_t>(s[i]);
  return g1;
}

__device__ __forceinline__ Descriptor2D make_descriptor_2d(
    uintptr_t lds_addr, uintptr_t global_addr, DataSize data_size, uint32_t tensor_dim0,
    uint32_t tensor_dim1, uint32_t tile_dim0, uint32_t tile_dim1, uint64_t tensor_dim0_stride,
    uint64_t tensor_dim1_stride, uint32_t workgroup_mask = 1) {
  Descriptor2D d;
  d.group0 = make_group0(lds_addr, global_addr);
  d.group1 = make_group1(data_size, tensor_dim0, tensor_dim1, tile_dim0, tile_dim1,
                         tensor_dim0_stride, tensor_dim1_stride, workgroup_mask);
  return d;
}

#if HIP_HAVE_TDM_INTRINSICS

// ==============================================================================
// RAW INTRINSIC WRAPPERS
// ==============================================================================

// Stalls the wave until fewer than Cnt+1 of its TDM ops are in flight;
// wait<0>() drains all of them. Cnt (and CachePolicy below) is a template
// parameter, not a plain argument, because the underlying builtins require
// an immarg (a literal at the call site).
template <uint16_t Cnt>
__device__ __forceinline__ void wait() {
  __builtin_amdgcn_s_wait_tensorcnt(Cnt);
}

template <int CachePolicy>
__device__ __forceinline__ void load_to_lds(u32x4 group0, i32x8 group1,
                                            i32x4 group2 = i32x4{0, 0, 0, 0},
                                            i32x4 group3 = i32x4{0, 0, 0, 0}) {
  const i32x8 group4{0, 0, 0, 0, 0, 0, 0, 0};  // reserved; always zero
  __builtin_amdgcn_tensor_load_to_lds(group0, group1, group2, group3, group4, CachePolicy);
}

template <int CachePolicy>
__device__ __forceinline__ void store_from_lds(u32x4 group0, i32x8 group1,
                                               i32x4 group2 = i32x4{0, 0, 0, 0},
                                               i32x4 group3 = i32x4{0, 0, 0, 0}) {
  const i32x8 group4{0, 0, 0, 0, 0, 0, 0, 0};  // reserved; always zero
  __builtin_amdgcn_tensor_store_from_lds(group0, group1, group2, group3, group4, CachePolicy);
}

template <int CachePolicy>
__device__ __forceinline__ void load_to_lds(const Descriptor2D &d) {
  load_to_lds<CachePolicy>(d.group0, d.group1);
}

template <int CachePolicy>
__device__ __forceinline__ void store_from_lds(const Descriptor2D &d) {
  store_from_lds<CachePolicy>(d.group0, d.group1);
}

#endif  // HIP_HAVE_TDM_INTRINSICS

// ==============================================================================
// BULK TILE-COPY HELPERS: flat global<->global byte copies staged through
// LDS, tiled as 2D descriptors with 8-byte elements. Must be called from
// exactly one lane; callers own that gating and any tail bytes left
// uncovered by covered_bytes().
// ==============================================================================

constexpr uint32_t FlatCopyElementLog2 = 3;  // 8 bytes/element

// Bytes of [0, total_bytes) actually moved for a given tile_bytes_cap: the
// largest multiple of tile_bytes_cap not exceeding total_bytes, or a single
// undersized tile if total_bytes is smaller than one.
__device__ __forceinline__ size_t covered_bytes(size_t total_bytes, uint32_t tile_bytes_cap) {
  if (total_bytes == 0) return 0;
  if (total_bytes < tile_bytes_cap) {
    constexpr size_t elem_bytes = size_t{1} << FlatCopyElementLog2;
    return total_bytes - (total_bytes % elem_bytes);
  }
  return (total_bytes / tile_bytes_cap) * static_cast<size_t>(tile_bytes_cap);
}

#if HIP_HAVE_TDM_INTRINSICS

// Copies covered_bytes(bytes, tile_bytes_cap) bytes tile by tile through
// lds_scratch. double_buffered overlaps load(tile N+1) with store(tile N)
// (needs 2*tile_bytes_cap bytes); otherwise each tile is a serialized
// load-wait-store-wait. tile_bytes_cap must be a multiple of 8 bytes.
//
// LoadCachePolicy/StoreCachePolicy are template parameters (see load_to_lds
// above), built with make_cache_policy(hint, scope).
template <int LoadCachePolicy, int StoreCachePolicy>
__device__ inline void copy_region(void *dst, const void *src, size_t bytes, void *lds_scratch,
                                   uint32_t tile_bytes_cap, bool double_buffered) {
  const size_t covered = covered_bytes(bytes, tile_bytes_cap);
  if (covered == 0) return;
  const uint32_t tile_bytes =
      (bytes < tile_bytes_cap) ? static_cast<uint32_t>(covered) : tile_bytes_cap;
  const size_t full_tiles = covered / tile_bytes;
  const uint32_t tile_dim0 = tile_bytes >> FlatCopyElementLog2;

  // Tensor layout is invariant across tiles — only the addresses (GROUP0)
  // change per iteration.
  Descriptor2D desc;
  desc.group1 = make_group1(DataSize::Bits64, /*tensor_dim0=*/tile_dim0, /*tensor_dim1=*/1,
                            /*tile_dim0=*/tile_dim0, /*tile_dim1=*/1,
                            /*tensor_dim0_stride=*/tile_dim0, /*tensor_dim1_stride=*/tile_dim0);

  auto *dst_bytes = static_cast<char *>(dst);
  const auto *src_bytes = static_cast<const char *>(src);
  auto *lds_base = static_cast<uint8_t *>(lds_scratch);

#if defined(__gfx1250__)
  if (double_buffered) {
    uintptr_t ping = reinterpret_cast<uintptr_t>(lds_base);
    uintptr_t pong = reinterpret_cast<uintptr_t>(lds_base + tile_bytes);

    // Prologue: start loading tile 0 into ping.
    desc.group0 = make_group0(ping, reinterpret_cast<uintptr_t>(src_bytes));
    load_to_lds<LoadCachePolicy>(desc);

    for (size_t i = 0; i < full_tiles; ++i) {
      const bool has_next = (i + 1) < full_tiles;

      // Wait for the current tile's load to land before storing from it.
      wait<0>();

      // Prefetch tile i+1 into pong and store tile i from ping — both
      // enqueued before the next wait, so the tensor engine runs them
      // concurrently.
      if (has_next) {
        desc.group0 =
            make_group0(pong, reinterpret_cast<uintptr_t>(src_bytes + (i + 1) * tile_bytes));
        load_to_lds<LoadCachePolicy>(desc);
      }
      desc.group0 = make_group0(ping, reinterpret_cast<uintptr_t>(dst_bytes + i * tile_bytes));
      store_from_lds<StoreCachePolicy>(desc);

      uintptr_t tmp = ping;
      ping = pong;
      pong = tmp;
    }
    // Drain the last in-flight load+store pair.
    wait<0>();
  } else
#endif
  {
    const uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_base);
    for (size_t i = 0; i < full_tiles; ++i) {
      const size_t offset = i * tile_bytes;

      desc.group0 = make_group0(lds_addr, reinterpret_cast<uintptr_t>(src_bytes + offset));
      load_to_lds<LoadCachePolicy>(desc);
      wait<0>();

      desc.group0 = make_group0(lds_addr, reinterpret_cast<uintptr_t>(dst_bytes + offset));
      store_from_lds<StoreCachePolicy>(desc);
      wait<0>();
    }
  }
}

// Same contract as copy_region, but always double-buffered and pipelined
// tighter: waits only for the in-flight count to drop to <=1 (wait<1>())
// instead of draining every iteration, so a load and a store are almost
// always both in flight. Tiles issue in pairs (load/store alternating
// even/odd, buffer chosen by parity); each op is issued exactly two slots
// after the op it depends on, so wait<1>() before op N guarantees op N-2
// has completed. Falls back to copy_region's single-buffered path on
// non-gfx1250 builds.
template <int LoadCachePolicy, int StoreCachePolicy>
__device__ inline void copy_region_pipelined(void *dst, const void *src, size_t bytes,
                                             void *lds_scratch, uint32_t tile_bytes_cap) {
#if defined(__gfx1250__)
  const size_t covered = covered_bytes(bytes, tile_bytes_cap);
  if (covered == 0) return;
  const uint32_t tile_bytes =
      (bytes < tile_bytes_cap) ? static_cast<uint32_t>(covered) : tile_bytes_cap;
  const size_t full_tiles = covered / tile_bytes;
  const uint32_t tile_dim0 = tile_bytes >> FlatCopyElementLog2;

  Descriptor2D desc;
  desc.group1 = make_group1(DataSize::Bits64, tile_dim0, 1, tile_dim0, 1, tile_dim0, tile_dim0);

  auto *dst_bytes = static_cast<char *>(dst);
  const auto *src_bytes = static_cast<const char *>(src);
  const uintptr_t buf[2] = {
      reinterpret_cast<uintptr_t>(lds_scratch),
      reinterpret_cast<uintptr_t>(static_cast<uint8_t *>(lds_scratch) + tile_bytes)};

  auto issue_load = [&](uintptr_t lds_addr, uintptr_t global_addr) {
    desc.group0 = make_group0(lds_addr, global_addr);
    load_to_lds<LoadCachePolicy>(desc);
  };
  auto issue_store = [&](uintptr_t lds_addr, uintptr_t global_addr) {
    desc.group0 = make_group0(lds_addr, global_addr);
    store_from_lds<StoreCachePolicy>(desc);
  };

  // Prologue: load tile 0 into buf[0].
  issue_load(buf[0], reinterpret_cast<uintptr_t>(src_bytes));

  for (size_t tile_even = 0; tile_even < full_tiles; tile_even += 2) {
    const size_t tile_odd = tile_even + 1;
    const bool has_odd = tile_odd < full_tiles;

    if (has_odd) {
      wait<1>();
      issue_load(buf[tile_odd & 1u], reinterpret_cast<uintptr_t>(src_bytes + tile_odd * tile_bytes));
    }

    // No odd tile: load(tile_even) was only one op ago, not two, so
    // wait<1>() would race ahead of it -- drain fully instead.
    if (has_odd) {
      wait<1>();
    } else {
      wait<0>();
    }
    issue_store(buf[tile_even & 1u], reinterpret_cast<uintptr_t>(dst_bytes + tile_even * tile_bytes));

    if (has_odd) {
      wait<1>();
      issue_store(buf[tile_odd & 1u], reinterpret_cast<uintptr_t>(dst_bytes + tile_odd * tile_bytes));

      const size_t next_even = tile_even + 2;
      if (next_even < full_tiles) {
        wait<1>();
        issue_load(buf[next_even & 1u],
                   reinterpret_cast<uintptr_t>(src_bytes + next_even * tile_bytes));
      }
    }
  }

  // Drain everything still in flight before returning.
  wait<0>();
#else
  copy_region<LoadCachePolicy, StoreCachePolicy>(dst, src, bytes, lds_scratch, tile_bytes_cap,
                          /*double_buffered=*/false);
#endif
}

#endif  // HIP_HAVE_TDM_INTRINSICS

// ==============================================================================
// LDS REGISTRATION: backs rocshmem_set_tdm_lds()/rocshmem_query_tdm_lds_bytes().
// Every thread in the block must call set_lds() -- only thread (0,0,0)'s
// arguments are used, but its internal barrier needs every thread to reach
// it -- before any work-group put/get in that kernel that should use TDM.
// ==============================================================================

// LDS bytes needed for a double-buffered copy_region call with this tile size.
__host__ __device__ constexpr size_t lds_bytes_for_tile(uint32_t tile_bytes) {
  return 2 * static_cast<size_t>(tile_bytes);
}

// Each workgroup gets its own private LDS, and set_lds() always runs earlier
// in the same kernel dispatch than any get_lds() that reads it back, so no
// staleness check is needed here.
struct LdsRegistration {
  void* ptr;
  size_t bytes;
};

inline __shared__ LdsRegistration lds_registration;

__device__ __forceinline__ void set_lds(void* ptr, size_t bytes) {
  if (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 && hipThreadIdx_z == 0) {
    lds_registration = LdsRegistration{ptr, bytes};
  }
  __builtin_amdgcn_s_barrier();  // publish to the rest of the block
}

__device__ __forceinline__ LdsRegistration get_lds() {
  return lds_registration;
}

}  // namespace tdm
}  // namespace rocshmem

#endif  // LIBRARY_SRC_TDM_HPP_
