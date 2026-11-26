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

#ifndef LIBRARY_SRC_ASSEMBLY_HPP_
#define LIBRARY_SRC_ASSEMBLY_HPP_

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

namespace rocshmem {

#define DO_PRAGMA(x) _Pragma(#x)
#define NOWARN(warnoption, ...)                 \
  DO_PRAGMA(GCC diagnostic push)                \
  DO_PRAGMA(GCC diagnostic ignored #warnoption) \
  __VA_ARGS__                                   \
  DO_PRAGMA(GCC diagnostic pop)

// XNACK-Replay (gfx1250/MI450): a memory instruction may re-execute until its
// address translation completes (tracked by XCNT). s_wait_loadcnt only waits
// for data, not translation
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
#define ROCSHMEM_XNACK_XCNT_WAIT "\ns_wait_xcnt 0x0"
// No __builtin_amdgcn_s_wait_xcnt (or llvm.amdgcn.s.wait.xcnt) exists, so
// atomic-builtin paths can't fold this into the load/store the way the raw
// asm below folds it into ROCSHMEM_XNACK_XCNT_WAIT; it must be issued by hand.
//
// The dummy "v"(val) input is required, not cosmetic: without a real data
// dependency on the loaded register, this asm has nothing tying it to the
// load, so the scheduler is free to hoist it above the compiler's own
// s_wait_loadcnt (which only guards the register's first real use) -- this
// was observed in practice, reordering xcnt before loadcnt relative to the
// asm path's audited loadcnt-then-xcnt order (commit f91f5a1a395,
// AIROCSHMEM-533). Making the fence consume val forces the backend's
// SIInsertWaitcnts pass to resolve the load (i.e. emit s_wait_loadcnt)
// before this asm, restoring that order.
template <typename U>
__device__ __forceinline__ void xnack_xcnt_fence(U val) {
  // Widen to 32 bits first: a bare 8-bit "v" operand fails register
  // allocation for this asm on this target; the widen still carries the
  // dependency since it is itself a real use of the hazardous register.
  uint32_t dep = static_cast<uint32_t>(val);
  asm volatile("s_wait_xcnt 0x0" :: "v"(dep) : "memory");
}
#else
#define ROCSHMEM_XNACK_XCNT_WAIT ""
template <typename U>
__device__ __forceinline__ void xnack_xcnt_fence(U) {}
#endif

// Relaxed, system-scope load/store through an explicit address-space(1)
// (global) pointer, so the compiler is guaranteed to emit global_atomic_*
// rather than flat_atomic_* regardless of whether it can prove the generic
// pointer's address space itself (see ASM_TO_ATOMICS.md risk #1; commits
// 9a4fbd27be0/4256a218b12 are prior incidents of flat vs. global mismatches).
// detail::atomic::{load,store} (atomic.hpp) can't be reused here: template
// deduction folds the address-space qualifier into T, and __scoped_atomic_*_n
// then rejects the address-space-qualified by-value local it builds T in.
// Callers must only use these on a genuine global address (symmetric heap,
// or an IPC/RDMA-mapped remote of it) -- the same precondition the asm below
// already assumes for anything but CachePolicy::Standard.
//
// KNOWN, ACCEPTED GAP (gfx90a / GFX11 only): __scoped_atomic_{load,store}_n
// on these two targets emits only the "glc" cache bit for every scope/order
// combination (verified by disassembling relaxed/acquire/seq_cst at both
// MEMORY_SCOPE_DEVICE and MEMORY_SCOPE_SYSTEM: all four produce "glc", never
// "glc slc"). The asm this replaces always used "glc slc" for SystemScope on
// these two targets, which additionally bypasses L2. gfx942/gfx950 ("sc0
// sc1") and GFX12/gfx1250[-strict] ("scope:SCOPE_SYS") both match the asm
// exactly and are unaffected. This was a deliberate, discussed call to
// proceed everywhere despite the gap rather than keep gfx90a/GFX11 on asm;
// validate L2-bypass behavior on real gfx90a/GFX11 hardware (not available
// in the environment this conversion was written in) before relying on it.
template <typename U>
__device__ __forceinline__ U system_relaxed_load(const U* src) {
  using GU = const U __attribute__((address_space(1)));
  // A C-style cast is required here: Clang permits a generic-to-addrspace(1)
  // addrspacecast only through a C-style (or static_cast-like) cast, not
  // through reinterpret_cast.
  U val = __scoped_atomic_load_n((GU*)src,
                                 __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  xnack_xcnt_fence(val);
  return val;
}

template <typename U>
__device__ __forceinline__ void system_relaxed_store(U* dst, U val) {
  using GU = U __attribute__((address_space(1)));
  __scoped_atomic_store_n((GU*)dst, val,
                          __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  xnack_xcnt_fence(val);
}

// read_value is a genuine global address (symmetric heap / IPC-mapped
// remote); assigned_value is local, so only the read needs the atomic.
__device__ __forceinline__ void refresh_volatile_sbyte(volatile int *assigned_value,
                                                       volatile char *read_value) {
  int8_t val = system_relaxed_load(
      reinterpret_cast<const int8_t*>(const_cast<char*>(read_value)));
  *assigned_value = static_cast<int>(val);
}

__device__ __forceinline__ void refresh_volatile_dwordx2(volatile uint64_t *assigned_value,
                                                         volatile uint64_t *read_value) {
  *assigned_value = system_relaxed_load(const_cast<const uint64_t*>(read_value));
}

template <typename T>
__device__ __forceinline__ T uncached_load([[maybe_unused]] T* src) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 ||
                sizeof(T) == 8 || sizeof(T) == 16,
                "uncached_load only supports 1/2/4/8/16-byte types");
  if constexpr (sizeof(T) == 1) {
    uint8_t val = system_relaxed_load(reinterpret_cast<const uint8_t*>(src));
    return __builtin_bit_cast(T, val);
  } else if constexpr (sizeof(T) == 2) {
    uint16_t val = system_relaxed_load(reinterpret_cast<const uint16_t*>(src));
    return __builtin_bit_cast(T, val);
  } else if constexpr (sizeof(T) == 4) {
    uint32_t val = system_relaxed_load(reinterpret_cast<const uint32_t*>(src));
    return __builtin_bit_cast(T, val);
  } else if constexpr (sizeof(T) == 8) {
    uint64_t val = system_relaxed_load(reinterpret_cast<const uint64_t*>(src));
    return __builtin_bit_cast(T, val);
  } else {
    // No native 128-bit atomic; keep the raw asm path (see
    // ASM_TO_ATOMICS.md §5 -- not a goal of this conversion).
    T ret{};
#if defined(__gfx90a__) || defined(__GFX11__)
    asm volatile(
        "global_load_dwordx4 %0 %1 off glc slc \n"
        "s_waitcnt vmcnt(0)"
        : "=&v"(ret)
        : "v"(src)
        : "memory");
#endif
#if defined(__gfx942__) || defined(__gfx950__)
    asm volatile(
        "global_load_dwordx4 %0 %1 off sc0 sc1 \n"
        "s_waitcnt vmcnt(0)"
        : "=&v"(ret)
        : "v"(src)
        : "memory");
#endif
#if defined(__GFX12__)
    asm volatile(
        "global_load_b128 %0 %1 off scope:SCOPE_SYS \n"
        "s_wait_loadcnt 0x0" ROCSHMEM_XNACK_XCNT_WAIT
        : "=&v"(ret)
        : "v"(src)
        : "memory");
#endif
    return ret;
  }
}

__device__ __forceinline__ void __roc_flush() {
  // No-op: HDP flush is handled elsewhere (see hdp_policy.hpp); there is no
  // cache-flush instruction sequence for any currently-supported arch here.
}

// ==============================================================================
// BUFFER RESOURCE INTRINSICS
// ==============================================================================

using i32x4_t = int32_t __attribute__((ext_vector_type(4)));

__device__ __uint128_t llvm_amdgcn_raw_buffer_load_b128(
    i32x4_t srsrc, uint32_t voffset, uint32_t soffset,
    uint32_t aux) __asm("llvm.amdgcn.raw.buffer.load.i128");

__device__ void llvm_amdgcn_raw_buffer_store_b128(
    __uint128_t vdata, i32x4_t srsrc, uint32_t voffset, uint32_t soffset,
    uint32_t aux) __asm("llvm.amdgcn.raw.buffer.store.i128");

__device__ uint64_t llvm_amdgcn_raw_buffer_load_b64(
    i32x4_t srsrc, uint32_t voffset, uint32_t soffset,
    uint32_t aux) __asm("llvm.amdgcn.raw.buffer.load.i64");

__device__ void llvm_amdgcn_raw_buffer_store_b64(
    uint64_t vdata, i32x4_t srsrc, uint32_t voffset, uint32_t soffset,
    uint32_t aux) __asm("llvm.amdgcn.raw.buffer.store.i64");

__device__ uint32_t llvm_amdgcn_raw_buffer_load_b32(
    i32x4_t srsrc, uint32_t voffset, uint32_t soffset,
    uint32_t aux) __asm("llvm.amdgcn.raw.buffer.load.i32");

__device__ void llvm_amdgcn_raw_buffer_store_b32(
    uint32_t vdata, i32x4_t srsrc, uint32_t voffset, uint32_t soffset,
    uint32_t aux) __asm("llvm.amdgcn.raw.buffer.store.i32");

__device__ uint16_t llvm_amdgcn_raw_buffer_load_b16(
    i32x4_t srsrc, uint32_t voffset, uint32_t soffset,
    uint32_t aux) __asm("llvm.amdgcn.raw.buffer.load.i16");

__device__ void llvm_amdgcn_raw_buffer_store_b16(
    uint16_t vdata, i32x4_t srsrc, uint32_t voffset, uint32_t soffset,
    uint32_t aux) __asm("llvm.amdgcn.raw.buffer.store.i16");

__device__ uint8_t llvm_amdgcn_raw_buffer_load_b8(
    i32x4_t srsrc, uint32_t voffset, uint32_t soffset,
    uint32_t aux) __asm("llvm.amdgcn.raw.buffer.load.i8");

__device__ void llvm_amdgcn_raw_buffer_store_b8(
    uint8_t vdata, i32x4_t srsrc, uint32_t voffset, uint32_t soffset,
    uint32_t aux) __asm("llvm.amdgcn.raw.buffer.store.i8");

// ==============================================================================
// CACHE POLICIES
// ==============================================================================
// GlobalCache, NonTemporal, DeviceScope and SystemScopeNT were removed: no
// caller anywhere in the tree (scalar AsmAccess load/store, load_buffer /
// store_buffer, or lds_stage_policy.hpp) ever instantiated them (see
// ASM_TO_ATOMICS.md §3); NonTemporal/SystemScopeNT also could not have been
// converted to atomics even if used, since the "nt" hint has no atomic-
// builtin equivalent (see ASM_TO_ATOMICS.md §5).
enum class CachePolicy {
  Standard,    // Normal C++ load/store (L1 and L2 cached)
  SystemScope  // Bypass L1 and L2 (sc0 sc1 / glc slc / scope:SYS)
};

// Maps a CachePolicy to the AMDGCN buffer instruction aux bits (sc0/sc1).
__host__ __device__ constexpr uint32_t cache_policy_aux(CachePolicy p) {
#if defined(__gfx942__) || defined(__gfx950__)
  return p == CachePolicy::SystemScope ? 0b10001 : 0b00000;  // sc1|sc0 : wave scope
#elif defined(__GFX12__)
  return p == CachePolicy::SystemScope ? 0b11000 : 0b00000;  // scope:SYS|temporal:RT : CU_RT
#else
  return p == CachePolicy::SystemScope ? 0b11 : 0b00;  // glc slc : none
#endif
}

__device__ __forceinline__ i32x4_t
make_buffer_resource(const void* base, uint32_t num_bytes) {
  uint64_t addr = reinterpret_cast<uint64_t>(base);
  i32x4_t rsrc;
  rsrc[0] = static_cast<int32_t>(addr & 0xFFFFFFFFu);
  rsrc[1] = static_cast<int32_t>(addr >> 32);
  rsrc[2] = static_cast<int32_t>(num_bytes);
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
  rsrc[3] = 0;  // raw buffer descriptor: gfx1250 requires a zeroed 3rd dword
                // (see CK_BUFFER_RESOURCE_3RD_DWORD in ck/ck.hpp)
#elif defined(__gfx1201__) || defined(__GFX11__)
  rsrc[3] = 0x31014000;  // raw buffer descriptor: no stride/swizzle
#else
  rsrc[3] = 0x00020000;  // raw buffer descriptor: no stride/swizzle
#endif
  return rsrc;
}

// Any policy but Standard emits global_load_*/global_store_* asm, which
// requires a genuine global address (symmetric heap, or an IPC/RDMA-mapped
// remote of it); Standard instead falls back to a plain load/store.
template <int Size, CachePolicy LoadPolicy = CachePolicy::Standard,
          CachePolicy StorePolicy = LoadPolicy>
struct AsmAccess;

// ==============================================================================
// 16-BYTE ACCESS (128-bit)
// ==============================================================================
template <CachePolicy LoadPolicy, CachePolicy StorePolicy>
struct AsmAccess<16, LoadPolicy, StorePolicy> {
  using type = __int128_t;

  // No native 128-bit atomic exists, so SystemScope keeps the raw asm here
  // regardless of policy count (see ASM_TO_ATOMICS.md §5 -- not a goal of
  // this conversion); only Standard is handled separately.
  static __device__ __forceinline__ type load(void* src) {
    if constexpr (LoadPolicy == CachePolicy::Standard) {
      return *reinterpret_cast<type*>(src);
    } else {
      type val{};
#if defined(__gfx942__) || defined(__gfx950__)
      asm volatile("global_load_dwordx4 %0 %1 off sc0 sc1\n"
                   "s_waitcnt vmcnt(0)" : "=&v"(val) : "v"(src) : "memory");
#elif defined(__gfx90a__) || defined(__GFX11__)
      asm volatile("global_load_dwordx4 %0 %1 off glc slc\n"
                   "s_waitcnt vmcnt(0)" : "=&v"(val) : "v"(src) : "memory");
#elif defined(__GFX12__)
      asm volatile("global_load_b128 %0 %1 off scope:SCOPE_SYS\n"
                   "s_wait_loadcnt 0x0" ROCSHMEM_XNACK_XCNT_WAIT
                   : "=&v"(val)
                   : "v"(src)
                   : "memory");
#else
      val = *reinterpret_cast<type*>(src);
#endif
      return val;
    }
  }

  static __device__ __forceinline__ void store(void* dst, type val) {
    if constexpr (StorePolicy == CachePolicy::Standard) {
      *reinterpret_cast<type*>(dst) = val;
    } else {
#if defined(__gfx942__) || defined(__gfx950__)
      asm volatile("global_store_dwordx4 %0 %1 off sc0 sc1" : : "v"(dst), "v"(val) : "memory");
#elif defined(__gfx90a__) || defined(__GFX11__)
      asm volatile("global_store_dwordx4 %0 %1 off glc slc" : : "v"(dst), "v"(val) : "memory");
#elif defined(__GFX12__)
      asm volatile("global_store_b128 %0 %1 off scope:SCOPE_SYS" ROCSHMEM_XNACK_XCNT_WAIT
                   :
                   : "v"(dst), "v"(val)
                   : "memory");
#else
      *reinterpret_cast<type*>(dst) = val;
#endif
    }
  }

  static __device__ __forceinline__ type load_buffer(const void *ptr,
                                                     uint32_t buf_size,
                                                     uint32_t offset) {
    return load_buffer(make_buffer_resource(ptr, buf_size), offset);
  }

  static __device__ __forceinline__ void store_buffer(void *ptr,
                                                      uint32_t buf_size,
                                                      uint32_t offset,
                                                      type val) {
    store_buffer(make_buffer_resource(ptr, buf_size), offset, val);
  }

  // rsrc-taking overloads: let callers hoist make_buffer_resource() out of a
  // loop so the compiler doesn't have to prove src/dst are wave-uniform on
  // every iteration (it can't, across a __noinline__ boundary, and falls
  // back to an expensive v_readfirstlane waterfall per call otherwise).
  static __device__ __forceinline__ type load_buffer(i32x4_t rsrc,
                                                     uint32_t offset) {
    constexpr uint32_t aux = cache_policy_aux(LoadPolicy);
    return __builtin_bit_cast(type,
        llvm_amdgcn_raw_buffer_load_b128(rsrc, offset, 0, aux));
  }

  static __device__ __forceinline__ void store_buffer(i32x4_t rsrc,
                                                      uint32_t offset,
                                                      type val) {
    constexpr uint32_t aux = cache_policy_aux(StorePolicy);
    llvm_amdgcn_raw_buffer_store_b128(static_cast<__uint128_t>(val),
                                      rsrc, offset, 0, aux);
  }
};

// ==============================================================================
// 8-BYTE ACCESS (64-bit)
// ==============================================================================
template <CachePolicy LoadPolicy, CachePolicy StorePolicy>
struct AsmAccess<8, LoadPolicy, StorePolicy> {
  using type = int64_t;

  static __device__ __forceinline__ type load(void* src) {
    if constexpr (LoadPolicy == CachePolicy::Standard) {
      return *reinterpret_cast<type*>(src);
    } else {
      static_assert(LoadPolicy == CachePolicy::SystemScope);
      uint64_t val = system_relaxed_load(reinterpret_cast<const uint64_t*>(src));
      return __builtin_bit_cast(type, val);
    }
  }

  static __device__ __forceinline__ void store(void* dst, type val) {
    if constexpr (StorePolicy == CachePolicy::Standard) {
      *reinterpret_cast<type*>(dst) = val;
    } else {
      static_assert(StorePolicy == CachePolicy::SystemScope);
      system_relaxed_store(reinterpret_cast<uint64_t*>(dst),
                           __builtin_bit_cast(uint64_t, val));
    }
  }

  static __device__ __forceinline__ type load_buffer(const void *ptr,
                                                     uint32_t buf_size,
                                                     uint32_t offset) {
    return load_buffer(make_buffer_resource(ptr, buf_size), offset);
  }

  static __device__ __forceinline__ void store_buffer(void *ptr,
                                                      uint32_t buf_size,
                                                      uint32_t offset,
                                                      type val) {
    store_buffer(make_buffer_resource(ptr, buf_size), offset, val);
  }

  static __device__ __forceinline__ type load_buffer(i32x4_t rsrc,
                                                     uint32_t offset) {
    constexpr uint32_t aux = cache_policy_aux(LoadPolicy);
    return __builtin_bit_cast(type,
        llvm_amdgcn_raw_buffer_load_b64(rsrc, offset, 0, aux));
  }

  static __device__ __forceinline__ void store_buffer(i32x4_t rsrc,
                                                      uint32_t offset,
                                                      type val) {
    constexpr uint32_t aux = cache_policy_aux(StorePolicy);
    llvm_amdgcn_raw_buffer_store_b64(static_cast<uint64_t>(val),
                                     rsrc, offset, 0, aux);
  }
};

// ==============================================================================
// 4-BYTE ACCESS (32-bit)
// ==============================================================================
template <CachePolicy LoadPolicy, CachePolicy StorePolicy>
struct AsmAccess<4, LoadPolicy, StorePolicy> {
  using type = int32_t;

  static __device__ __forceinline__ type load(void* src) {
    if constexpr (LoadPolicy == CachePolicy::Standard) {
      return *reinterpret_cast<type*>(src);
    } else {
      static_assert(LoadPolicy == CachePolicy::SystemScope);
      uint32_t val = system_relaxed_load(reinterpret_cast<const uint32_t*>(src));
      return __builtin_bit_cast(type, val);
    }
  }

  static __device__ __forceinline__ void store(void* dst, type val) {
    if constexpr (StorePolicy == CachePolicy::Standard) {
      *reinterpret_cast<type*>(dst) = val;
    } else {
      static_assert(StorePolicy == CachePolicy::SystemScope);
      system_relaxed_store(reinterpret_cast<uint32_t*>(dst),
                           __builtin_bit_cast(uint32_t, val));
    }
  }

  static __device__ __forceinline__ type load_buffer(const void *ptr,
                                                     uint32_t buf_size,
                                                     uint32_t offset) {
    return load_buffer(make_buffer_resource(ptr, buf_size), offset);
  }

  static __device__ __forceinline__ void store_buffer(void *ptr,
                                                      uint32_t buf_size,
                                                      uint32_t offset,
                                                      type val) {
    store_buffer(make_buffer_resource(ptr, buf_size), offset, val);
  }

  static __device__ __forceinline__ type load_buffer(i32x4_t rsrc,
                                                     uint32_t offset) {
    constexpr uint32_t aux = cache_policy_aux(LoadPolicy);
    return __builtin_bit_cast(type,
        llvm_amdgcn_raw_buffer_load_b32(rsrc, offset, 0, aux));
  }

  static __device__ __forceinline__ void store_buffer(i32x4_t rsrc,
                                                      uint32_t offset,
                                                      type val) {
    constexpr uint32_t aux = cache_policy_aux(StorePolicy);
    llvm_amdgcn_raw_buffer_store_b32(static_cast<uint32_t>(val),
                                     rsrc, offset, 0, aux);
  }
};

// ==============================================================================
// 2-BYTE ACCESS (16-bit / Widened)
// ==============================================================================
template <CachePolicy LoadPolicy, CachePolicy StorePolicy>
struct AsmAccess<2, LoadPolicy, StorePolicy> {
  using type = int16_t;

  static __device__ __forceinline__ type load(void* src) {
    if constexpr (LoadPolicy == CachePolicy::Standard) {
      return *reinterpret_cast<type*>(src);
    } else {
      static_assert(LoadPolicy == CachePolicy::SystemScope);
      uint16_t val = system_relaxed_load(reinterpret_cast<const uint16_t*>(src));
      return __builtin_bit_cast(type, val);
    }
  }

  static __device__ __forceinline__ void store(void* dst, type val) {
    if constexpr (StorePolicy == CachePolicy::Standard) {
      *reinterpret_cast<type*>(dst) = val;
    } else {
      static_assert(StorePolicy == CachePolicy::SystemScope);
      system_relaxed_store(reinterpret_cast<uint16_t*>(dst),
                           __builtin_bit_cast(uint16_t, val));
    }
  }

  static __device__ __forceinline__ type load_buffer(const void *ptr,
                                                     uint32_t buf_size,
                                                     uint32_t offset) {
    return load_buffer(make_buffer_resource(ptr, buf_size), offset);
  }

  static __device__ __forceinline__ void store_buffer(void *ptr,
                                                      uint32_t buf_size,
                                                      uint32_t offset,
                                                      type val) {
    store_buffer(make_buffer_resource(ptr, buf_size), offset, val);
  }

  static __device__ __forceinline__ type load_buffer(i32x4_t rsrc,
                                                     uint32_t offset) {
    constexpr uint32_t aux = cache_policy_aux(LoadPolicy);
    return static_cast<type>(
        llvm_amdgcn_raw_buffer_load_b16(rsrc, offset, 0, aux));
  }

  static __device__ __forceinline__ void store_buffer(i32x4_t rsrc,
                                                      uint32_t offset,
                                                      type val) {
    constexpr uint32_t aux = cache_policy_aux(StorePolicy);
    llvm_amdgcn_raw_buffer_store_b16(static_cast<uint16_t>(val),
                                     rsrc, offset, 0, aux);
  }
};

// ==============================================================================
// 1-BYTE ACCESS (8-bit / Widened)
// ==============================================================================
template <CachePolicy LoadPolicy, CachePolicy StorePolicy>
struct AsmAccess<1, LoadPolicy, StorePolicy> {
  using type = uint8_t;

  static __device__ __forceinline__ type load(void* src) {
    if constexpr (LoadPolicy == CachePolicy::Standard) {
      return *reinterpret_cast<type*>(src);
    } else {
      static_assert(LoadPolicy == CachePolicy::SystemScope);
      return system_relaxed_load(reinterpret_cast<const uint8_t*>(src));
    }
  }

  static __device__ __forceinline__ void store(void* dst, type val) {
    if constexpr (StorePolicy == CachePolicy::Standard) {
      *reinterpret_cast<type*>(dst) = val;
    } else {
      static_assert(StorePolicy == CachePolicy::SystemScope);
      system_relaxed_store(reinterpret_cast<uint8_t*>(dst), val);
    }
  }

  static __device__ __forceinline__ type load_buffer(const void *ptr,
                                                     uint32_t buf_size,
                                                     uint32_t offset) {
    return load_buffer(make_buffer_resource(ptr, buf_size), offset);
  }

  static __device__ __forceinline__ void store_buffer(void *ptr,
                                                      uint32_t buf_size,
                                                      uint32_t offset,
                                                      type val) {
    store_buffer(make_buffer_resource(ptr, buf_size), offset, val);
  }

  static __device__ __forceinline__ type load_buffer(i32x4_t rsrc,
                                                     uint32_t offset) {
    constexpr uint32_t aux = cache_policy_aux(LoadPolicy);
    return static_cast<type>(
        llvm_amdgcn_raw_buffer_load_b8(rsrc, offset, 0, aux));
  }

  static __device__ __forceinline__ void store_buffer(i32x4_t rsrc,
                                                      uint32_t offset,
                                                      type val) {
    constexpr uint32_t aux = cache_policy_aux(StorePolicy);
    llvm_amdgcn_raw_buffer_store_b8(val, rsrc, offset, 0, aux);
  }
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_ASSEMBLY_HPP_
