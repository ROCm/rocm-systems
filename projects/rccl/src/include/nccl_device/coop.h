/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_COOP_H_
#define _NCCL_DEVICE_COOP_H_
#include "utility.h"

// ncclCoop[Foo]: NCCL's versions of CUDA's Cooperative Groups. They conform
// to just this subset of the CUDA API:
//   int Coop::thread_rank();
//   int Coop::size();
//   int Coop::num_threads(); // same as size()
//   void Coop::sync();

#if defined(__HIP_DEVICE_COMPILE__)
  #if defined(__GFX9__)
  #define WARP_SIZE 64
  #else
  #define WARP_SIZE 32
  #endif
#else
  #define WARP_SIZE 32
#endif

#if NCCL_CHECK_CUDACC
#if defined(__HIP_PLATFORM_AMD__)
using ncclCoopMask_t = uint64_t;
static constexpr ncclCoopMask_t ncclCoopFullMask = ~0ull;
NCCL_DEVICE_INLINE int ncclCoopPopc(ncclCoopMask_t x) { return (int)__popcll(x); }
#else
using ncclCoopMask_t = uint32_t;
static constexpr ncclCoopMask_t ncclCoopFullMask = ~0u;
NCCL_DEVICE_INLINE int ncclCoopPopc(ncclCoopMask_t x) { return (int)__popc(x); }
#endif
#endif

#if NCCL_CHECK_CUDACC
struct ncclCoopAny {
  struct Storage { alignas(alignof(void*)) char space[16]; };
  struct VTable {
    int(*thread_rank)(void const*);
    int(*size)(void const*);
    void(*sync)(void*);
  };

  template<typename Impl>
  __device__ static int thread_rank(void const* o) {
    return static_cast<Impl const*>(o)->thread_rank();
  }
  template<typename Impl>
  __device__ static int size(void const* o) {
    return static_cast<Impl const*>(o)->size();
  }
  template<typename Impl>
  __device__ static void sync(void* o) {
    static_cast<Impl*>(o)->sync();
  }

  template<typename Impl>
  __device__ static VTable const* get_vtable() {
    static_assert(sizeof(Impl) <= sizeof(Storage), "Incompatible coop type size");
    static_assert(alignof(Impl) <= alignof(Storage), "Incompatible coop type alignment");
    static constexpr VTable v = {
      &thread_rank<Impl>,
      &size<Impl>,
      &sync<Impl>
    };
    return &v;
  }

  Storage storage;
  VTable const* vtable;

  ncclCoopAny(ncclCoopAny const&) = default;
  ncclCoopAny(ncclCoopAny&&) = default;
  ncclCoopAny() = default;

  template<typename Impl>
  __device__ ncclCoopAny(Impl impl) {
    ::new (&this->storage) Impl(impl);
    this->vtable = get_vtable<Impl>();
  }

  __device__ int thread_rank() const { return vtable->thread_rank(&storage); }
  __device__ int size() const { return vtable->size(&storage); }
  __device__ int num_threads() const { return vtable->size(&storage); }
  __device__ void sync() { vtable->sync(&storage); }
};
#endif

#if NCCL_CHECK_CUDACC
template<int nThreadsPow2>
struct ncclCoopTile { // An aligned pow2 set of threads within the warp.
  static_assert(nccl::utility::isPow2(nThreadsPow2) && nThreadsPow2 <= WARP_SIZE, "Condition required");

  NCCL_DEVICE_INLINE int thread_rank() const {
    return nccl::utility::lane() % nThreadsPow2;
  }
  NCCL_DEVICE_INLINE constexpr int size() const { return nThreadsPow2; }
  NCCL_DEVICE_INLINE constexpr int num_threads() const { return nThreadsPow2; }

  NCCL_DEVICE_INLINE ncclCoopMask_t laneMask() const {
    return (ncclCoopMask_t(-1)>>(WARP_SIZE-nThreadsPow2))<<(nccl::utility::lane() & -nThreadsPow2);
  }
  NCCL_DEVICE_INLINE void sync() {
#if ROCM_VERSION >= 70000
    if (nThreadsPow2 > 1) __syncwarp(laneMask());
#else
    __syncthreads();
#endif
  }
};
#endif

#if NCCL_CHECK_CUDACC
typedef ncclCoopTile<1> ncclCoopThread;
typedef ncclCoopTile<WARP_SIZE> ncclCoopWarp;
#endif

#if NCCL_CHECK_CUDACC
struct ncclCoopLanes { // Some lanes of this warp.
  ncclCoopMask_t lmask;

  NCCL_DEVICE_INLINE constexpr ncclCoopLanes(ncclCoopMask_t lmask = ncclCoopFullMask): lmask(lmask) {}

  NCCL_DEVICE_INLINE int thread_rank() const {
    return ncclCoopPopc(lmask & static_cast<ncclCoopMask_t>(nccl::utility::lanemask_lt()));
  }
  NCCL_DEVICE_INLINE int size() const {
    return ncclCoopPopc(lmask);
  }
  NCCL_DEVICE_INLINE int num_threads() const {
    return ncclCoopPopc(lmask);
  }
  NCCL_DEVICE_INLINE void sync() {
#if ROCM_VERSION >= 70000
    __syncwarp(lmask);
#else
    __syncthreads();
#endif
  }
};
#endif

#if NCCL_CHECK_CUDACC
#if defined(__HIP_PLATFORM_AMD__)
// AMD has no named barriers, so emulate sub-block warp-span sync with a shared,
// sense-reversing barrier keyed by span id. Init must zero the slots before use.
constexpr int ncclCoopNamedBarrierSlots = 16; // mirrors CUDA's 16 hardware named barriers (ids 0-15)
struct ncclCoopNamedBarrierSlot { uint32_t arrive; uint32_t sense; };

#if !defined(__clang_llvm_bitcode_lib__)
// Native/in-tree path: function-local LDS. It is statically reachable from the
// kernel that inlines this, so the backend sizes it into the kernel's group
// segment. Compiled out of the bitcode library, where the only path to the
// LDS-using sync() is an indirect (vtable) call the LDS-sizing pass can't see.
NCCL_DEVICE_INLINE ncclCoopNamedBarrierSlot* ncclCoopNamedBarrierState() {
  __shared__ ncclCoopNamedBarrierSlot slots[ncclCoopNamedBarrierSlots];
  return slots;
}
#endif

// Zero caller-provided (kernel-owned) slots before first use. The bitcode
// binding passes a kernel __shared__ pointer here so the LDS lives in the
// launching kernel rather than in the indirectly-called thunk.
NCCL_DEVICE_INLINE void ncclCoopNamedBarrierInit(ncclCoopNamedBarrierSlot* slots) {
  for (int i = threadIdx.x; i < ncclCoopNamedBarrierSlots; i += blockDim.x) { slots[i].arrive = 0; slots[i].sense = 0; }
  __syncthreads();
}

#if !defined(__clang_llvm_bitcode_lib__)
// Back-compat no-arg form: existing in-tree kernels keep calling this unchanged.
NCCL_DEVICE_INLINE void ncclCoopNamedBarrierInit() {
  ncclCoopNamedBarrierInit(ncclCoopNamedBarrierState());
}
#endif
#else
NCCL_DEVICE_INLINE void ncclCoopNamedBarrierInit() {}
#endif

struct ncclCoopWarpSpan {
  uint32_t warp0:8, nWarps:8, id:8;
#if defined(__HIP_PLATFORM_AMD__)
  // Caller/kernel-owned named-barrier scratch. Optional on native AMD (sync()
  // falls back to function-local LDS); supplied by the bitcode thunk, where a
  // function-local __shared__ in this indirectly-called code can't be sized
  // into the launching kernel's group segment (HSA exception 0x1016 otherwise).
  ncclCoopNamedBarrierSlot* slots;

  NCCL_DEVICE_INLINE constexpr ncclCoopWarpSpan(int warp0, int nWarps, int id,
                                                ncclCoopNamedBarrierSlot* slots = nullptr):
    warp0(warp0), nWarps(nWarps), id(id), slots(slots) {
  }
#else
  NCCL_DEVICE_INLINE constexpr ncclCoopWarpSpan(int warp0, int nWarps, int id):
    warp0(warp0), nWarps(nWarps), id(id) {
  }
#endif

  NCCL_DEVICE_INLINE int thread_rank() const {
    return threadIdx.x - WARP_SIZE*warp0;
  }
  NCCL_DEVICE_INLINE int size() const {
    return WARP_SIZE*nWarps;
  }
  NCCL_DEVICE_INLINE int num_threads() const {
    return WARP_SIZE*nWarps;
  }

  NCCL_DEVICE_INLINE void sync() {
  #if defined(__HIP_PLATFORM_AMD__)
    // __syncthreads() can't sync a subset of warps; emulate a named barrier in software.
    // Single-warp span is lockstep: skip the shared atomic (hot path) and just fence.
    using Atom = cuda::atomic_ref<uint32_t, cuda::thread_scope_block>;
    if (nWarps <= 1) { cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_block); return; }
  #if defined(__clang_llvm_bitcode_lib__)
    ncclCoopNamedBarrierSlot* slot = &slots[id];                          // bitcode: caller-owned LDS only
  #else
    ncclCoopNamedBarrierSlot* slot = &(slots ? slots : ncclCoopNamedBarrierState())[id];
  #endif
    cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_block);
    if ((threadIdx.x % WARP_SIZE) == 0) {  // one leader per warp
      uint32_t s = Atom{slot->sense}.load(cuda::memory_order_relaxed);
      if (Atom{slot->arrive}.fetch_add(1u, cuda::memory_order_relaxed) + 1u == (uint32_t)nWarps) {
        Atom{slot->arrive}.store(0u, cuda::memory_order_relaxed);  // last in: reset, flip to release
        Atom{slot->sense}.store(s ^ 1u, cuda::memory_order_release);
      } else {
        while (Atom{slot->sense}.load(cuda::memory_order_acquire) == s)
          __builtin_amdgcn_s_sleep(1);
      }
    }
    cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_block);
  #else
    asm volatile("barrier.sync %0, %1;" :: "r"(1+id), "r"(32*nWarps) : "memory");
    __barrier_sync_count(1+id, 32*nWarps);
#endif
  }
};
#if defined(__HIP_PLATFORM_AMD__)
// The added scratch pointer must keep ncclCoopWarpSpan within ncclCoopAny's
// type-erased Storage (16 bytes); this is tight (4B bitfield + 8B pointer).
static_assert(sizeof(ncclCoopWarpSpan) <= 16, "ncclCoopWarpSpan must fit ncclCoopAny::Storage");
#endif
#endif

#if NCCL_CHECK_CUDACC
struct ncclCoopCta {
  NCCL_DEVICE_INLINE int thread_rank() const { return threadIdx.x; }
  NCCL_DEVICE_INLINE int size() const { return blockDim.x; }
  NCCL_DEVICE_INLINE int num_threads() const { return blockDim.x; }
  NCCL_DEVICE_INLINE void sync() { __syncthreads(); }
};
#endif

#if NCCL_CHECK_CUDACC
// NOTE: upstream v2.30 renamed ncclCoopLaneMask -> ncclCoopGetLaneMask; the
// auto-merged callers (impl/vector__funcs.h, impl/core__funcs.h) use the new
// name, so adopt it here while keeping RCCL's wave64 ncclCoopMask_t bodies.
template<int nThreadsPow2>
NCCL_DEVICE_INLINE ncclCoopMask_t ncclCoopGetLaneMask(ncclCoopTile<nThreadsPow2> coop) {
  return coop.laneMask();
}
NCCL_DEVICE_INLINE ncclCoopMask_t ncclCoopGetLaneMask(ncclCoopLanes coop) {
  return coop.lmask;
}
NCCL_DEVICE_INLINE ncclCoopMask_t ncclCoopGetLaneMask(ncclCoopWarpSpan coop) {
  return ncclCoopFullMask;
}
NCCL_DEVICE_INLINE ncclCoopMask_t ncclCoopGetLaneMask(ncclCoopCta coop) {
  return ncclCoopFullMask;
}
#endif

#if NCCL_CHECK_CUDACC
template<int nThreads>
NCCL_DEVICE_INLINE constexpr bool ncclCoopIsThread(ncclCoopTile<nThreads>) {
  return nThreads == 1;
}
NCCL_DEVICE_INLINE constexpr bool ncclCoopIsThread(ncclCoopLanes) { return false; }
NCCL_DEVICE_INLINE constexpr bool ncclCoopIsThread(ncclCoopWarpSpan) { return false; }
NCCL_DEVICE_INLINE constexpr bool ncclCoopIsThread(ncclCoopCta) { return false; }
#endif

#if NCCL_CHECK_CUDACC
template<int nThreads>
NCCL_DEVICE_INLINE constexpr bool ncclCoopWithinWarp(ncclCoopTile<nThreads>) { return true; }
NCCL_DEVICE_INLINE constexpr bool ncclCoopWithinWarp(ncclCoopLanes) { return true; }
NCCL_DEVICE_INLINE constexpr bool ncclCoopWithinWarp(ncclCoopWarpSpan) { return false; }
NCCL_DEVICE_INLINE constexpr bool ncclCoopWithinWarp(ncclCoopCta) { return false; }
#endif

#if NCCL_CHECK_CUDACC
// Pick threads of our warp that are safe to use collectively.
NCCL_DEVICE_INLINE ncclCoopLanes ncclCoopCoalesced() {
#if ROCM_VERSION >= 70000
  return ncclCoopLanes{static_cast<ncclCoopMask_t>(__activemask())};
#else
  return ncclCoopLanes{static_cast<ncclCoopMask_t>(__ballot(1))};
#endif
}
#endif

#if NCCL_CHECK_CUDACC
// Pick threads of our warp that are safe to use collectively given that this
// is a collective on the provided cooperative group.
template<typename Coop>
NCCL_DEVICE_INLINE ncclCoopTile<WARP_SIZE> ncclCoopCoalesced(Coop) {
  return ncclCoopTile<WARP_SIZE>();
}
NCCL_DEVICE_INLINE ncclCoopLanes ncclCoopCoalesced(ncclCoopLanes coop) {
  return coop;
}
template<int nThreads>
NCCL_DEVICE_INLINE ncclCoopTile<nThreads> ncclCoopCoalesced(ncclCoopTile<nThreads> coop) {
  return coop;
}
#endif

#if NCCL_CHECK_CUDACC
template<int nThreads, typename T>
NCCL_DEVICE_INLINE T ncclCoopBcast(ncclCoopTile<nThreads>, T value, int root, bool entrySync=true) {
  constexpr int n = (sizeof(T)+4-1)/4;
  union { uint32_t u[n]; T v; };
  v = value;
  #pragma unroll
  for (int i=0; i < n; i++) u[i] = __shfl_sync(-1u, u[i], root, nThreads);
  return v;
}
template<typename T>
NCCL_DEVICE_INLINE T ncclCoopBcast(ncclCoopLanes coop, T value, int root, bool entrySync=true) {
  uint32_t m = coop.lmask;
  uint32_t r = root == 0 ? __ffs(m)-1 : __fns(m, 0, 1+root);
  constexpr int n = (sizeof(T)+4-1)/4;
  union { uint32_t u[n]; T v; };
  v = value;
  #pragma unroll
  for (int i=0; i < n; i++) u[i] = __shfl_sync(m, u[i], r);
  return v;
}

NCCL_DEVICE_INLINE ulong2* ncclCoopBcast_WarpSpan_stash() {
  __shared__ ulong2 stash[15];
  return stash;
}

template<typename T>
NCCL_DEVICE_INLINE T ncclCoopBcast(ncclCoopWarpSpan coop, T value, int root, bool entrySync=true) {
  static_assert(sizeof(T) <= sizeof(ncclCoopBcast_WarpSpan_stash()[0]), "Required");
  if (entrySync) coop.sync();
  if (coop.thread_rank() == root) *(T*)&ncclCoopBcast_WarpSpan_stash()[coop.id] = value;
  coop.sync();
  return *(T*)&ncclCoopBcast_WarpSpan_stash()[coop.id];
}

NCCL_DEVICE_INLINE ulong2* ncclCoopBcast_Cta_stash() {
  __shared__ ulong2 stash;
  return &stash;
}

template<typename T>
NCCL_DEVICE_INLINE T ncclCoopBcast(ncclCoopCta coop, T value, int root, bool entrySync=true) {
  static_assert(sizeof(T) <= sizeof(*ncclCoopBcast_Cta_stash()), "Required");
  if (entrySync) coop.sync();
  if (coop.thread_rank() == root) *(T*)ncclCoopBcast_Cta_stash() = value;
  coop.sync();
  return *(T*)ncclCoopBcast_Cta_stash();
}
#endif

#endif
